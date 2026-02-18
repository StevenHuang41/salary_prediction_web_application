import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest';
import MyCarousel from '../MyCarousel';

const testImgs = [
  'img1.png',
  'img2.png',
  null,
];

const testAlts = [
  '1 images',
  null,
  '3 images',
];

describe('MyCarousel', () => {
  it('render nothing if images and its alts are null', () => {
    render(<MyCarousel images={null} alts={null}/>)
    expect(document.querySelector('img')).not.toBeInTheDocument();
  });

  it('render images and its alts in carousel', () => {
    render(<MyCarousel images={testImgs} alts={testAlts}/>)
    const n_images = screen.getAllByRole('img');
    expect(n_images.length).toBe(testImgs.length);
  });

  it('images with null src or alt when inputs have null value', () => {
    render(<MyCarousel images={testImgs} alts={testAlts}/>)
    const images = document.querySelectorAll('img');
    expect(images[1]).not.toHaveAttribute('alt');
    expect(images[2]).not.toHaveAttribute('src');
  })

  it('handle empty images array', () => {
    render(<MyCarousel images={[]} alts={[]}/>);
    expect(document.querySelector('img')).not.toBeInTheDocument();
  });

});